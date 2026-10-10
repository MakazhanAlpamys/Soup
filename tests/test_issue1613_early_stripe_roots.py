"""#1613 — Early validation for SOUP_LAYER_STREAM_STRIPE_DIRS.

Pinned by this suite:
- A CliRunner test of soup train with stream_layers: true and the variable naming
  a folder that does not exist: exit code 1, output contains "must be an existing
  directory", does not contain "Loaded:" or "Run ID", and ExperimentTracker.start_run
  was not called.
- The same config with --dry-run: the same refusal, and no "Config valid" line.
- A same-volume entry is refused at the same early point.
- When stream_layers is off/unset in config, the variable is not checked.
- soup doctor with the variable set to a missing folder prints a row that names
  the entry and the rule; with the variable unset the output is unchanged.
- soup doctor with --disk probes each stripe root and reports NVMe.
- soup doctor never prints raw control characters from an entry.
- soup doctor count cap reports MAX_STREAM_READ_AHEAD.
- soup doctor refuses a stripe root on the same volume as the primary cache root.
- Output assertions go through ANSI-strip and whitespace-collapse helper, and pass
  with FORCE_COLOR=1.
"""

from __future__ import annotations

import json
import os
import re
from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.config_bounds import MAX_STREAM_READ_AHEAD
from soup_cli.utils.stripe_roots import (
    MAX_STRIPE_DIRS,
    STRIPE_DIRS_ENV,
    StripeRootError,
    iter_early_stripe_roots,
    validate_early_stripe_roots,
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str | None) -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(_ANSI.sub("", text or "").replace("│", " ").split())


def _doctor_resources_raw(monkeypatch, *, probe_disk: bool = False) -> str:
    from soup_cli.commands import doctor

    out = StringIO()
    monkeypatch.setattr(doctor, "console", Console(file=out, width=400, color_system=None))
    doctor._check_resources(probe_disk=probe_disk)
    return out.getvalue()


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

        missing_name = "missing_stripe_folder"
        missing_dir = str(tmp_path / missing_name)
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
        assert missing_name in output.replace(" ", ""), output
        assert "Loaded:" not in output, output
        assert "Run ID" not in output, output
        assert "Config valid" not in output, output
        assert "StripeRootError:" not in output, output
        assert not start_run_mock.called

    def test_same_volume_entry_refused_at_same_early_point(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        start_run_mock = MagicMock()
        monkeypatch.setattr(ExperimentTracker, "start_run", start_run_mock)

        cache_dir = tmp_path / "cache_dir"
        cache_dir.mkdir()
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(cache_dir))

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

        result = CliRunner().invoke(app, ["train", "--config", cfg_file, "--dry-run"])
        output = _plain(result.output)

        assert "must be an existing directory" not in output
        assert "same volume as the primary" not in output

    def test_variable_is_not_checked_when_stream_layers_is_off(self, tmp_path, monkeypatch):
        missing = str(tmp_path / "does-not-exist-xyz")
        monkeypatch.setenv(STRIPE_DIRS_ENV, missing)
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(
            "base: smollm2-135m\ntask: sft\nbackend: transformers\n"
            "data:\n  train: /dev/null\ntraining:\n  batch_size: 1\n"
        )
        result = CliRunner().invoke(app, ["train", "--config", str(cfg), "--dry-run"])
        output = _plain(result.output)
        assert "Dry run" in output, (output, repr(result.exception))
        assert "must be an existing directory" not in output, output

    def test_dry_run_with_a_real_dataset_stops_at_the_refusal(self, tmp_path, monkeypatch):
        data = tmp_path / "train.jsonl"
        data.write_text(
            "".join(
                json.dumps({"instruction": f"q{idx}", "input": "", "output": f"a{idx}"}) + "\n"
                for idx in range(4)
            ),
            encoding="utf-8",
        )
        cfg = tmp_path / "soup.yaml"
        cfg.write_text(
            "base: smollm2-135m\ntask: sft\nbackend: transformers\n"
            f"data:\n  train: {data.as_posix()}\n"
            "training:\n  stream_layers: true\n  batch_size: 1\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(tmp_path / "missing_stripe_folder"))
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(app, ["train", "--config", str(cfg), "--dry-run"])
        output = _plain(result.output)

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "must be an existing directory" in output, output
        assert "validating data" not in output, output
        assert "Config valid" not in output, output


class TestDoctorStripeOutput:
    def test_doctor_names_a_missing_stripe_folder_and_the_rule(self, tmp_path, monkeypatch):
        missing = str(tmp_path / "does-not-exist-xyz")
        monkeypatch.setenv(STRIPE_DIRS_ENV, missing)
        output = _plain(_doctor_resources_raw(monkeypatch))
        assert missing in output, output
        assert "must be an existing directory" in output, output

    def test_doctor_output_unchanged_when_variable_unset(self, monkeypatch):
        monkeypatch.delenv(STRIPE_DIRS_ENV, raising=False)
        output_unset = _plain(_doctor_resources_raw(monkeypatch))
        assert "Stripe root" not in output_unset

    def test_doctor_disk_flag_probes_each_stripe_root(self, tmp_path, monkeypatch):
        import soup_cli.utils.stripe_roots as stripe_roots

        folder = tmp_path / "stripe"
        folder.mkdir()
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(folder))
        real = os.path.realpath(str(folder))
        monkeypatch.setattr(
            stripe_roots, "volume_of", lambda p: 2 if os.path.realpath(p) == real else 1
        )
        probed = []

        def fake_kind(path):
            probed.append(path)
            return "nvme"

        monkeypatch.setattr("soup_cli.utils.layer_stream.detect_disk_kind", fake_kind)
        output = _plain(_doctor_resources_raw(monkeypatch, probe_disk=True))
        assert real in probed, probed
        assert f"Stripe root {folder} — NVMe" in output, output

    def test_doctor_never_prints_a_control_character_of_an_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(tmp_path / "stripe\x1b[31mred"))
        raw = _doctor_resources_raw(monkeypatch)
        assert "\x1b" not in raw, ascii(raw)
        assert "contains control characters" in _plain(raw), raw

    def test_doctor_count_cap_names_the_real_read_ahead_limit(self, tmp_path, monkeypatch):
        entries = [str(tmp_path / f"s{n}") for n in range(MAX_STRIPE_DIRS + 1)]
        monkeypatch.setenv(STRIPE_DIRS_ENV, os.pathsep.join(entries))
        output = _plain(_doctor_resources_raw(monkeypatch))
        assert f"stops at {MAX_STREAM_READ_AHEAD}," in output, output

    def test_doctor_refuses_a_stripe_folder_on_the_primary_volume(self, tmp_path, monkeypatch):
        (tmp_path / "cache").mkdir()
        (tmp_path / "stripe").mkdir()
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(tmp_path / "stripe"))
        output = _plain(_doctor_resources_raw(monkeypatch))
        assert "on the same volume as the primary cache root" in output, output

    def test_doctor_disk_flag_refuses_a_stripe_root_that_is_not_nvme(self, tmp_path, monkeypatch):
        import soup_cli.utils.stripe_roots as stripe_roots
        from soup_cli.commands import doctor

        folder = tmp_path / "stripe"
        folder.mkdir()
        monkeypatch.setenv(STRIPE_DIRS_ENV, str(folder))
        real = os.path.realpath(str(folder))
        monkeypatch.setattr(
            stripe_roots, "volume_of", lambda p: 2 if os.path.realpath(p) == real else 1
        )
        monkeypatch.setattr("soup_cli.utils.layer_stream.detect_disk_kind", lambda path: "ssd")
        out = StringIO()
        monkeypatch.setattr(doctor, "console", Console(file=out, width=400, color_system=None))
        issues = []
        doctor._check_resources(probe_disk=True, issues=issues)
        output = _plain(out.getvalue())
        assert f"Stripe root {folder} — Refused" in output, output
        assert "classifies as 'ssd'" in output, output
        assert len(issues) == 1 and STRIPE_DIRS_ENV in issues[0], issues

    def test_doctor_records_an_issue_for_a_refused_entry_without_disk(self, tmp_path, monkeypatch):
        from soup_cli.commands import doctor

        missing = str(tmp_path / "does-not-exist-xyz")
        monkeypatch.setenv(STRIPE_DIRS_ENV, missing)
        out = StringIO()
        monkeypatch.setattr(doctor, "console", Console(file=out, width=400, color_system=None))
        issues = []
        doctor._check_resources(probe_disk=False, issues=issues)
        assert "must be an existing directory" in _plain(out.getvalue()), out.getvalue()
        assert len(issues) == 1 and STRIPE_DIRS_ENV in issues[0], issues


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

    def test_iter_early_stripe_roots_helper(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        stripe = tmp_path / "stripe"
        stripe.mkdir()

        results = list(
            iter_early_stripe_roots(
                str(cache), [str(stripe), str(tmp_path / "nonexistent")]
            )
        )
        assert len(results) == 2
        # stripe on same volume
        assert results[0][0] == str(stripe)
        assert results[0][1] is None
        assert "on the same volume" in results[0][2]

        # nonexistent
        assert results[1][0] == str(tmp_path / "nonexistent")
        assert results[1][1] is None
        assert "must be an existing directory" in results[1][2]
