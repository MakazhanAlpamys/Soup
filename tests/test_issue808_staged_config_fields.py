"""Staged config fields must warn in v0.76 and refuse in v0.77 (#808).

28 config fields are declared in schema.py for features that have not landed,
accepted silently, and read by nothing. Following the #627 warn-then-refuse
pattern across two releases:
- In v0.76 each staged field prints:
    <field> is accepted but read by nothing; v0.77 will refuse it
- The deadline version lives in ONE constant (STAGED_FIELD_REJECTION_VERSION).
- TestTheDeadline asserts against soup_cli.__version__ in both directions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from soup_cli import __version__
from soup_cli.config import loader
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


def _raw(extra_data: str = "", extra_training: str = "") -> dict:
    import yaml

    doc = yaml.safe_load(_VALID)
    if extra_data:
        doc["data"].update(yaml.safe_load(extra_data))
    if extra_training:
        doc["training"] = yaml.safe_load(extra_training)
    return doc


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
        """Derived, not duplicated: a second copy is what falls out of step."""
        version = re.escape(STAGED_FIELD_REJECTION_VERSION)
        pattern = re.compile(rf"""v{version}\b|["']{version}["']""")
        src = Path(__file__).parents[1] / "src" / "soup_cli"
        holders = sorted(
            p.relative_to(src).as_posix()
            for p in src.rglob("*.py")
            if p.name != "__init__.py" and pattern.search(p.read_text(encoding="utf-8"))
        )
        assert holders == ["config/staged_fields.py"], (
            f"the deadline version is written out in more than one place: {holders}"
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
    """Validate inventory size and default handling across all 28 staged fields."""

    def test_exactly_28_staged_fields_registered(self) -> None:
        assert len(STAGED_FIELDS) == 28, (
            f"Expected exactly 28 staged fields, got {len(STAGED_FIELDS)}"
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
            ("training", "convergence_window", 100),
            ("training", "convergence_rel_tol", 0.01),
            ("training", "forgetting_eval_steps", 50),
            ("training", "forgetting_threshold", 0.05),
            ("training", "forgetting_benchmark", "mini_common_sense"),
            ("training", "forgetting_stop", True),
            ("training", "checkpoint_eval_steps", 100),
            ("training", "checkpoint_eval_metric", "judge"),
            ("training", "checkpoint_eval_tasks", "eval.jsonl"),
            ("training", "checkpoint_keep_top", 5),
            ("training", "early_stop_patience", 4),
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

    def test_schema_defaults_produce_zero_staged_findings(self) -> None:
        raw = {"training": {}, "data": {}}
        for (sec, name), default_val in STAGED_FIELDS.items():
            raw[sec][name] = default_val
        found = find_staged_config_fields(raw)
        assert found == [], f"Default values should not trigger warnings: {found}"


class TestLoaderStagedFieldIntegration:
    """Test loader behavior under warning vs error severity."""

    def test_staged_field_warns_and_loads_under_warn_severity(self, capsys) -> None:
        yaml_str = _VALID + "\ntraining:\n  convergence_window: 100\n"
        cfg = loader.load_config_from_string(yaml_str)
        assert cfg.training.convergence_window == 100
        captured = capsys.readouterr()
        assert "training.convergence_window" in captured.out
        assert "accepted but read by nothing" in captured.out
        assert STAGED_FIELD_REJECTION_VERSION in captured.out
        assert "will refuse it" in captured.out

    def test_staged_field_refuses_under_error_severity(self, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        yaml_str = _VALID + "\ntraining:\n  convergence_window: 100\n"
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "training.convergence_window is accepted but read by nothing" in msg
        assert f"v{STAGED_FIELD_REJECTION_VERSION}" in msg

    def test_multiple_staged_fields_reported_together(self, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        yaml_str = (
            _VALID
            + "\ntraining:\n  convergence_window: 100\n  convergence_rel_tol: 0.01\n"
        )
        with pytest.raises(ValueError) as excinfo:
            loader.load_config_from_string(yaml_str)
        msg = str(excinfo.value)
        assert "training.convergence_window is accepted but read by nothing" in msg
        assert "training.convergence_rel_tol is accepted but read by nothing" in msg

    def test_load_config_file_warns(self, tmp_path, capsys) -> None:
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(_VALID + "\ntraining:\n  convergence_window: 100\n")
        cfg = loader.load_config(cfg_file)
        assert cfg.training.convergence_window == 100
        captured = capsys.readouterr()
        assert "training.convergence_window" in captured.out
        assert "accepted but read by nothing" in captured.out
        assert STAGED_FIELD_REJECTION_VERSION in captured.out
        assert "will refuse it" in captured.out

    def test_load_config_file_refuses_under_error(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", "error")
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(_VALID + "\ntraining:\n  convergence_window: 100\n")
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(cfg_file)
        assert excinfo.value.code == 1
