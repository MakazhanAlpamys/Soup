"""Tests for v0.43.0 Part D — soup data demo bundles."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.commands.data import app as data_app
from soup_cli.utils.demo_bundles import (
    DEMO_BUNDLE_NAMES,
    DemoBundle,
    copy_bundle_to,
    get_bundle,
    list_bundles,
)


class TestDemoBundleRegistry:
    def test_known_names(self):
        for name in ("alpaca_demo", "sharegpt_demo", "dpo_demo", "grpo_demo"):
            assert name in DEMO_BUNDLE_NAMES

    def test_immutable_set(self):
        with pytest.raises(AttributeError):
            DEMO_BUNDLE_NAMES.add("evil")  # type: ignore[attr-defined]

    def test_list_bundles_sorted(self):
        bundles = list_bundles()
        names = [b.name for b in bundles]
        assert names == sorted(names)

    def test_bundle_dataclass_frozen(self):
        b = list_bundles()[0]
        assert isinstance(b, DemoBundle)
        with pytest.raises(Exception):
            b.name = "evil"  # type: ignore[misc]

    def test_get_bundle_unknown(self):
        with pytest.raises(ValueError, match="unknown bundle"):
            get_bundle("garbage")

    def test_get_bundle_empty(self):
        with pytest.raises(ValueError, match="not be empty"):
            get_bundle("")

    def test_get_bundle_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            get_bundle("alpaca\x00")

    def test_get_bundle_oversize(self):
        with pytest.raises(ValueError, match="exceeds max"):
            get_bundle("a" * 64)

    def test_get_bundle_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_bundle(123)  # type: ignore[arg-type]


class TestCopyBundleTo:
    def test_writes_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = copy_bundle_to("alpaca_demo", "./alpaca_demo.jsonl")
        assert Path(out).is_file()
        # Validate every line parses as JSON.
        with open(out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        evil = str(tmp_path.parent / "evil.jsonl")
        with pytest.raises(ValueError, match="under cwd"):
            copy_bundle_to("alpaca_demo", evil)

    def test_existing_file_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "out.jsonl"
        target.write_text("placeholder", encoding="utf-8")
        with pytest.raises(FileExistsError):
            copy_bundle_to("alpaca_demo", str(target))

    def test_null_byte_output_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="null"):
            copy_bundle_to("alpaca_demo", "ev\x00il.jsonl")

    def test_non_string_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="must be a non-empty string"):
            copy_bundle_to("alpaca_demo", "")

    def test_unknown_bundle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="unknown bundle"):
            copy_bundle_to("garbage", "./out.jsonl")

    def test_symlink_at_tmp_path_rejected(self, tmp_path, monkeypatch):
        if os.name == "nt":
            pytest.skip("symlink creation needs admin/dev mode on Windows")
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "out.jsonl"
        # Pre-place a symlink at <target>.tmp pointing to a sentinel file.
        outside = tmp_path / "outside.txt"
        outside.write_text("placeholder", encoding="utf-8")
        os.symlink(str(outside), str(target) + ".tmp")
        with pytest.raises(ValueError, match="symlink"):
            copy_bundle_to("alpaca_demo", str(target))

    @pytest.mark.parametrize("name", sorted(DEMO_BUNDLE_NAMES))
    def test_every_bundle_copyable(self, name, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = copy_bundle_to(name, f"./{name}.jsonl")
        assert os.path.getsize(out) > 0


class TestDataDemoCli:
    def test_list_help(self):
        runner = CliRunner()
        result = runner.invoke(data_app, ["demo", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "demo bundle" in result.output.lower() or "bundle" in result.output.lower()

    def test_list_no_args(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(data_app, ["demo"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Table renders all 4 bundles.
        for name in ("alpaca_demo", "sharegpt_demo", "dpo_demo", "grpo_demo"):
            assert name in result.output

    def test_copy_default_output(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(data_app, ["demo", "alpaca_demo"])
            assert result.exit_code == 0, (result.output, repr(result.exception))
            assert Path("alpaca_demo.jsonl").is_file()

    def test_copy_custom_output(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(
                data_app, ["demo", "dpo_demo", "--output", "./mine.jsonl"]
            )
            assert result.exit_code == 0, (result.output, repr(result.exception))
            assert Path("mine.jsonl").is_file()

    def test_unknown_bundle_exits_2(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(data_app, ["demo", "garbage"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "unknown bundle" in result.output.lower()

    def test_existing_output_exits_1(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            Path("alpaca_demo.jsonl").write_text("x", encoding="utf-8")
            result = runner.invoke(data_app, ["demo", "alpaca_demo"])
            assert result.exit_code == 1, (result.output, repr(result.exception))
            assert "already exists" in result.output.lower()

    def test_outside_cwd_output_exits_1(self, tmp_path):
        runner = CliRunner()
        evil = str(tmp_path.parent / "evil.jsonl")
        with runner.isolated_filesystem(temp_dir=str(tmp_path)):
            result = runner.invoke(
                data_app, ["demo", "alpaca_demo", "--output", evil]
            )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "under cwd" in result.output.lower()


class TestCopyBundleEdgeCases:
    def test_size_cap_branch(self, tmp_path, monkeypatch):
        from soup_cli.utils import demo_bundles as db

        monkeypatch.chdir(tmp_path)
        big_src = tmp_path / "big.jsonl"
        line = json.dumps({"x": "y" * 1024}) + "\n"
        with open(big_src, "w", encoding="utf-8") as f:
            for _ in range(55_000):  # ~55 MB
                f.write(line)
        monkeypatch.setattr(db, "_bundle_source_path", lambda _b: str(big_src))
        with pytest.raises(ValueError, match="byte cap"):
            db.copy_bundle_to("alpaca_demo", "./out.jsonl")
        # Staged temp file must not survive the rejection.
        assert not (tmp_path / "out.jsonl.tmp").exists()
        assert not (tmp_path / "out.jsonl").exists()

    def test_invalid_json_line(self, tmp_path, monkeypatch):
        from soup_cli.utils import demo_bundles as db

        monkeypatch.chdir(tmp_path)
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
        monkeypatch.setattr(db, "_bundle_source_path", lambda _b: str(bad))
        with pytest.raises(ValueError, match="not valid JSON"):
            db.copy_bundle_to("alpaca_demo", "./out.jsonl")
        assert not (tmp_path / "out.jsonl").exists()
