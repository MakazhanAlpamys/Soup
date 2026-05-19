"""Tests for v0.61.0 Part E — `soup edit diff` (knowledge-injection diff).

Coverage:
- ``DiffReport`` / ``FactChange`` frozen dataclasses.
- ``load_probes`` JSONL loader (cwd-contained, symlink-rejected).
- ``build_diff_report`` schema-only orchestrator.
- ``render_diff_table`` + ``write_diff_report``.
- ``soup edit diff`` CLI smoke.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


class TestDataclasses:
    def test_imports(self):
        from soup_cli.utils.edit_diff import (
            DiffReport,
            FactChange,
            build_diff_report,
            load_probes,
            render_diff_table,
            write_diff_report,
        )
        assert callable(build_diff_report)
        assert callable(load_probes)
        assert callable(render_diff_table)
        assert callable(write_diff_report)
        assert dataclasses.is_dataclass(DiffReport)
        assert dataclasses.is_dataclass(FactChange)

    def test_fact_change_frozen(self):
        from soup_cli.utils.edit_diff import FactChange

        c = FactChange(
            prompt="p", before="b", after="a", changed=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.prompt = "x"  # type: ignore

    def test_diff_report_frozen(self):
        from soup_cli.utils.edit_diff import DiffReport, FactChange

        r = DiffReport(
            before_run_id="b",
            after_run_id="a",
            changes=(FactChange(prompt="p", before="b", after="a", changed=False),),
            total_probes=1,
            soup_version="0.61.0",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.total_probes = 999  # type: ignore

    def test_diff_report_validates_changes_tuple(self):
        from soup_cli.utils.edit_diff import DiffReport

        with pytest.raises(TypeError):
            DiffReport(
                before_run_id="b",
                after_run_id="a",
                changes=[],  # list, not tuple
                total_probes=0,
                soup_version="0.61.0",
            )

    def test_diff_report_to_dict(self):
        from soup_cli.utils.edit_diff import DiffReport, FactChange

        r = DiffReport(
            before_run_id="b",
            after_run_id="a",
            changes=(
                FactChange(prompt="p1", before="x", after="y", changed=True),
            ),
            total_probes=1,
            soup_version="0.61.0",
        )
        d = r.to_dict()
        assert d["before_run_id"] == "b"
        assert d["after_run_id"] == "a"
        assert d["total_probes"] == 1
        assert d["changes"][0]["prompt"] == "p1"
        # JSON round-trip
        assert json.loads(json.dumps(d)) == d

    def test_diff_report_total_probes_bool_rejected(self):
        from soup_cli.utils.edit_diff import DiffReport

        with pytest.raises(TypeError):
            DiffReport(
                before_run_id="b",
                after_run_id="a",
                changes=(),
                total_probes=True,  # type: ignore
                soup_version="0.61.0",
            )


class TestLoadProbes:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def test_happy_path(self, tmp_path):
        from soup_cli.utils.edit_diff import load_probes

        p = tmp_path / "probes.jsonl"
        self._write_jsonl(p, [{"prompt": "Who is X?"}, {"prompt": "Who is Y?"}])

        # Need to be in cwd
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            probes = load_probes("probes.jsonl")
        finally:
            os.chdir(old)
        assert probes == ("Who is X?", "Who is Y?")

    def test_missing_file(self, tmp_path):
        from soup_cli.utils.edit_diff import load_probes

        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(FileNotFoundError):
                load_probes("missing.jsonl")
        finally:
            os.chdir(old)

    def test_skips_malformed_rows(self, tmp_path):
        from soup_cli.utils.edit_diff import load_probes

        p = tmp_path / "probes.jsonl"
        p.write_text(
            '{"prompt": "good"}\n'
            'this is not json\n'
            '{"prompt": ""}\n'
            '{"prompt": "another good"}\n',
            encoding="utf-8",
        )

        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            probes = load_probes("probes.jsonl")
        finally:
            os.chdir(old)
        assert probes == ("good", "another good")

    def test_outside_cwd_rejected(self, tmp_path):
        from soup_cli.utils.edit_diff import load_probes

        out = tmp_path / "outside.jsonl"
        out.write_text('{"prompt": "p"}', encoding="utf-8")

        # cwd is the original; tmp_path is unrelated
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        old = os.getcwd()
        os.chdir(cwd_dir)
        try:
            with pytest.raises(ValueError, match="cwd"):
                load_probes(str(out))
        finally:
            os.chdir(old)

    def test_null_byte_rejected(self):
        from soup_cli.utils.edit_diff import load_probes

        with pytest.raises(ValueError):
            load_probes("probe\x00.jsonl")

    def test_empty_path_rejected(self):
        from soup_cli.utils.edit_diff import load_probes

        with pytest.raises(ValueError):
            load_probes("")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only symlink test")
    def test_symlink_rejected(self, tmp_path):
        from soup_cli.utils.edit_diff import load_probes

        real = tmp_path / "real.jsonl"
        real.write_text('{"prompt": "p"}', encoding="utf-8")
        link = tmp_path / "link.jsonl"
        os.symlink(real, link)

        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(ValueError, match="symlink"):
                load_probes("link.jsonl")
        finally:
            os.chdir(old)


class TestBuildDiffReport:
    def test_happy_path(self):
        from soup_cli.utils.edit_diff import build_diff_report

        report = build_diff_report(
            before_run_id="before-run",
            after_run_id="after-run",
        )
        assert report.before_run_id == "before-run"
        assert report.after_run_id == "after-run"
        assert report.changes == ()
        assert report.total_probes == 0

    def test_same_run_id_rejected(self):
        from soup_cli.utils.edit_diff import build_diff_report

        with pytest.raises(ValueError, match="differ"):
            build_diff_report(
                before_run_id="x",
                after_run_id="x",
            )

    def test_bool_run_id_rejected(self):
        from soup_cli.utils.edit_diff import build_diff_report

        with pytest.raises(TypeError):
            build_diff_report(
                before_run_id=True,  # type: ignore
                after_run_id="x",
            )

    def test_top_k_bool_rejected(self):
        from soup_cli.utils.edit_diff import build_diff_report

        with pytest.raises(TypeError):
            build_diff_report(
                before_run_id="b",
                after_run_id="a",
                top_k=True,  # type: ignore
            )

    def test_top_k_out_of_range(self):
        from soup_cli.utils.edit_diff import build_diff_report

        with pytest.raises(ValueError):
            build_diff_report(
                before_run_id="b", after_run_id="a", top_k=0,
            )
        with pytest.raises(ValueError):
            build_diff_report(
                before_run_id="b", after_run_id="a", top_k=200,
            )

    def test_with_probe_file(self, tmp_path):
        from soup_cli.utils.edit_diff import build_diff_report

        p = tmp_path / "probes.jsonl"
        p.write_text(
            '{"prompt": "q1"}\n{"prompt": "q2"}\n{"prompt": "q3"}\n',
            encoding="utf-8",
        )
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            r = build_diff_report(
                before_run_id="b",
                after_run_id="a",
                probe_file="probes.jsonl",
                top_k=2,
            )
        finally:
            os.chdir(old)
        assert r.total_probes == 3
        assert len(r.changes) == 2  # capped by top_k


class TestRenderDiffTable:
    def test_renders(self, capsys):
        from rich.console import Console

        from soup_cli.utils.edit_diff import (
            DiffReport,
            FactChange,
            render_diff_table,
        )

        r = DiffReport(
            before_run_id="b",
            after_run_id="a",
            changes=(
                FactChange(prompt="p1", before="x", after="y", changed=True),
            ),
            total_probes=1,
            soup_version="0.61.0",
        )
        console = Console(force_terminal=False, width=200)
        render_diff_table(r, console)
        # No raise = pass.

    def test_non_report_rejected(self):
        from rich.console import Console

        from soup_cli.utils.edit_diff import render_diff_table

        with pytest.raises(TypeError):
            render_diff_table("not a report", Console())  # type: ignore


class TestWriteDiffReport:
    def test_happy_path(self, tmp_path):
        from soup_cli.utils.edit_diff import (
            DiffReport,
            write_diff_report,
        )

        r = DiffReport(
            before_run_id="b",
            after_run_id="a",
            changes=(),
            total_probes=0,
            soup_version="0.61.0",
        )
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            out = "report.json"
            write_diff_report(r, out)
            assert Path(out).exists()
            data = json.loads(Path(out).read_text())
            assert data["before_run_id"] == "b"
        finally:
            os.chdir(old)

    def test_outside_cwd_rejected(self, tmp_path):
        from soup_cli.utils.edit_diff import (
            DiffReport,
            write_diff_report,
        )

        r = DiffReport(
            before_run_id="b",
            after_run_id="a",
            changes=(),
            total_probes=0,
            soup_version="0.61.0",
        )
        out = tmp_path / "outside.json"
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        old = os.getcwd()
        os.chdir(cwd_dir)
        try:
            with pytest.raises(ValueError):
                write_diff_report(r, str(out))
        finally:
            os.chdir(old)

    def test_non_report_rejected(self, tmp_path):
        from soup_cli.utils.edit_diff import write_diff_report

        with pytest.raises(TypeError):
            write_diff_report("not a report", "out.json")  # type: ignore


class TestCli:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["edit", "diff", "--help"])
        assert result.exit_code == 0, result.output

    def test_basic_diff(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [
                "edit", "diff", "before-run", "after-run",
            ])
            assert result.exit_code == 0, result.output

    def test_same_runs_rejected(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(app, [
                "edit", "diff", "same", "same",
            ])
            assert result.exit_code != 0

    def test_writes_output(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            out = Path(fs) / "diff.json"
            result = runner.invoke(app, [
                "edit", "diff", "before-run", "after-run",
                "--output", str(out),
            ])
            assert result.exit_code == 0, result.output
            assert out.exists()
            data = json.loads(out.read_text())
            assert data["total_probes"] == 0
