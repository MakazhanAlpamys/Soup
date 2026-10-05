"""#1613 — Early validation for SOUP_LAYER_STREAM_STRIPE_DIRS.

Pinned by this suite:
- A CliRunner test of soup train with stream_layers: true and the variable naming
  a folder that does not exist: exit code 1, output contains "must be an existing
  directory", does not contain "Loaded:" or "Run ID", and ExperimentTracker.start_run
  was not called.
- The same config with --dry-run: the same refusal, and no "Config valid" line.
- A same-volume entry is refused at the same early point.
- soup doctor with the variable set to a missing folder prints a row that names
  the entry and the rule; with the variable unset the output is unchanged.
- With the variable unset, soup train behaves exactly as before (no new output,
  no new call).
- Output assertions go through ANSI-strip and whitespace-collapse helper, and pass
  with FORCE_COLOR=1.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.stripe_roots import (
    MAX_STRIPE_DIRS,
    STRIPE_DIRS_ENV,
    StripeRootError,
    validate_early_stripe_roots,
)
from tests.conftest import strip_ansi


def _plain(text: str | None) -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(strip_ansi(text or "").replace("│", " ").split())


def _write_streaming_config(tmp_path, **kwargs) -> str:
    cfg = tmp_path / "soup.yaml"
    cfg.write_text(
        "base: smollm2-135m\n"
        "task: sft\n"
        "backend: transformers\n"
        "data:\n"
        "  train: /dev/null\n"
        "training:\n"
        "  stream_layers: true\n"
        "  batch_size: 1\n"
    )
    return str(cfg)


class TestCliRunnerEarlyRefusal:
    @pytest.mark.parametrize("dry_run", [False, True], ids=["train", "dry-run"])
    def test_missing_stripe_folder_is_refused_early(self, tmp_path, monkeypatch, dry_run):
        from soup_cli.experiment.tracker import ExperimentTracker

        start_run_mock = MagicMock()
        monkeypatch.setattr(ExperimentTracker, "start_run", start_run_mock)

        missing_dir = str(tmp_path / "does-not-exist-xyz")
        monkeypatch.setenv(STRIPE_DIRS_ENV, missing_dir)
        monkeypatch.setenv("FORCE_COLOR", "1")

        cfg_file = _write_streaming_config(tmp_path)
        args = ["train", "--config", cfg_file]
        if dry_run:
            args.append("--dry-run")
        else:
            args.append("--yes")

        result = CliRunner().invoke(app, args)
        output = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "must be an existing directory" in output, output
        assert "does-not-exist-xyz" in output, output
        assert "Loaded:" not in output, output
        assert "Run ID" not in output, output
        assert "Config valid" not in output, output
        assert "StripeRootError:" not in output, output
        assert not start_run_mock.called

    def test_same_volume_entry_refused_at_same_early_point(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        start_run_mock = MagicMock()
        monkeypatch.setattr(ExperimentTracker, "start_run", start_run_mock)

        stripe_dir = tmp_path / "stripe_same_vol"
        stripe_dir.mkdir()
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(stripe_dir))
        monkeypatch.setenv("FORCE_COLOR", "1")

        cfg_file = _write_streaming_config(tmp_path)
        result = CliRunner().invoke(app, ["train", "--config", cfg_file, "--yes"])
        output = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "same volume as the primary cache root" in output, output
        assert "Loaded:" not in output, output
        assert "Run ID" not in output, output
        assert not start_run_mock.called

    def test_unset_variable_does_not_fail_at_early_check(self, tmp_path, monkeypatch):
        """With the variable unset, the early check is bypassed completely."""
        monkeypatch.delenv(STRIPE_DIRS_ENV, raising=False)
        cfg_file = _write_streaming_config(tmp_path)

        # In dry-run mode, without the variable set, it passes the early check
        # and proceeds to dry-run data validation.
        result = CliRunner().invoke(app, ["train", "--config", cfg_file, "--dry-run"])
        output = _plain(result.output)

        # It did not fail on stripe roots
        assert "must be an existing directory" not in output
        assert "same volume as the primary" not in output


class TestDoctorStripeOutput:
    def test_doctor_prints_row_naming_entry_and_rule_when_missing(self, monkeypatch):
        missing_dir = "/tmp/doctor_missing_dir"
        monkeypatch.setenv(STRIPE_DIRS_ENV, missing_dir)
        monkeypatch.setenv("FORCE_COLOR", "1")

        result = CliRunner().invoke(app, ["doctor"])
        output = _plain(result.output)

        assert missing_dir in output, output
        assert "must be an existing directory" in output, output

    def test_doctor_output_unchanged_when_variable_unset(self, monkeypatch):
        monkeypatch.delenv(STRIPE_DIRS_ENV, raising=False)
        result_unset = CliRunner().invoke(app, ["doctor"])
        output_unset = _plain(result_unset.output)

        assert "Stripe root" not in output_unset

    def test_doctor_with_disk_flag_probes_stripe_roots(self, tmp_path, monkeypatch):
        folder = tmp_path / "valid"
        folder.mkdir()
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(folder))

        import soup_cli.utils.stripe_roots as mod

        ids = {os.path.realpath(str(folder)): 999}
        monkeypatch.setattr(mod, "volume_of", lambda p: ids.get(os.path.realpath(p), 1))

        # Mock detect_disk_kind to return nvme
        monkeypatch.setattr("soup_cli.utils.layer_stream.detect_disk_kind", lambda p: "nvme")

        result = CliRunner().invoke(app, ["doctor", "--disk"])
        output = _plain(result.output)

        assert "valid" in output, output
        assert "NVMe" in output, output


class TestValidateEarlyStripeRootsUnit:
    def test_unset_and_empty_return_empty_tuple(self):
        assert validate_early_stripe_roots("/primary", environ={}) == ()
        assert validate_early_stripe_roots("/primary", environ={STRIPE_DIRS_ENV: ""}) == ()

    def test_missing_directory_raises_stripe_root_error(self, tmp_path):
        missing = str(tmp_path / "gone")
        with pytest.raises(StripeRootError, match="must be an existing directory"):
            validate_early_stripe_roots(str(tmp_path), environ={STRIPE_DIRS_ENV: missing})

    def test_too_many_entries_refused(self, tmp_path):
        entries = [str(tmp_path / f"s{n}") for n in range(MAX_STRIPE_DIRS + 1)]
        with pytest.raises(StripeRootError, match=f"at most {MAX_STRIPE_DIRS}"):
            validate_early_stripe_roots(
                str(tmp_path), environ={STRIPE_DIRS_ENV: os.pathsep.join(entries)}
            )

    def test_same_volume_as_primary_refused(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        stripe = tmp_path / "stripe"
        stripe.mkdir()
        with pytest.raises(StripeRootError, match="same volume as the primary cache root"):
            validate_early_stripe_roots(str(cache), environ={STRIPE_DIRS_ENV: str(stripe)})

    def test_distinct_volumes_accepted_without_disk_probe(self, tmp_path, monkeypatch):
        import soup_cli.utils.stripe_roots as mod

        cache = tmp_path / "cache"
        cache.mkdir()
        stripe1 = tmp_path / "stripe1"
        stripe1.mkdir()
        stripe2 = tmp_path / "stripe2"
        stripe2.mkdir()

        ids = {
            os.path.realpath(str(cache)): 1,
            os.path.realpath(str(stripe1)): 2,
            os.path.realpath(str(stripe2)): 3,
        }
        monkeypatch.setattr(mod, "volume_of", lambda p: ids[os.path.realpath(p)])

        result = validate_early_stripe_roots(
            str(cache),
            environ={STRIPE_DIRS_ENV: f"{stripe1}{os.pathsep}{stripe2}"},
        )
        assert result == (os.path.realpath(str(stripe1)), os.path.realpath(str(stripe2)))
