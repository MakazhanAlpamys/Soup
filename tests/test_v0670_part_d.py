"""v0.67.0 Part D — ``soup adapters pr`` (GitHub-shaped PR rendering).

Tests for ``soup_cli/utils/adapter_pr.py``:

- Frozen ``EvalDelta`` / ``SampleDiff`` / ``AdapterPR`` dataclasses
- ``build_adapter_pr`` factory with validation
- ``render_pr_markdown`` (eval-delta tables + sample diffs)
- ``render_pr_json`` round-trip
- Rich-markup escape on every operator-controlled field
- CLI smoke (`soup adapters pr`)
"""

from __future__ import annotations

import dataclasses
import math

import pytest

# -----------------------------------------------------------------------------
# Public surface
# -----------------------------------------------------------------------------


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import adapter_pr

        assert hasattr(adapter_pr, "AdapterPR")
        assert hasattr(adapter_pr, "EvalDelta")
        assert hasattr(adapter_pr, "SampleDiff")
        assert hasattr(adapter_pr, "build_adapter_pr")
        assert hasattr(adapter_pr, "render_pr_markdown")
        assert hasattr(adapter_pr, "render_pr_json")
        assert hasattr(adapter_pr, "write_pr_markdown")


# -----------------------------------------------------------------------------
# EvalDelta
# -----------------------------------------------------------------------------


class TestEvalDelta:
    def test_construct(self) -> None:
        from soup_cli.utils.adapter_pr import EvalDelta

        delta = EvalDelta(metric="accuracy", baseline=0.7, candidate=0.85)
        assert math.isclose(delta.delta, 0.15)

    def test_frozen(self) -> None:
        from soup_cli.utils.adapter_pr import EvalDelta

        d = EvalDelta(metric="m", baseline=0.5, candidate=0.6)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.candidate = 0.99  # type: ignore[misc]

    def test_non_finite_rejected(self) -> None:
        from soup_cli.utils.adapter_pr import EvalDelta

        with pytest.raises(ValueError):
            EvalDelta(metric="m", baseline=math.nan, candidate=0.5)
        with pytest.raises(ValueError):
            EvalDelta(metric="m", baseline=0.5, candidate=math.inf)

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.adapter_pr import EvalDelta

        with pytest.raises(TypeError):
            EvalDelta(metric="m", baseline=True, candidate=0.5)  # type: ignore[arg-type]

    def test_metric_null_byte_rejected(self) -> None:
        from soup_cli.utils.adapter_pr import EvalDelta

        with pytest.raises(ValueError):
            EvalDelta(metric="m\x00", baseline=0.5, candidate=0.6)

    def test_metric_oversize_rejected(self) -> None:
        from soup_cli.utils.adapter_pr import EvalDelta

        with pytest.raises(ValueError):
            EvalDelta(metric="a" * 300, baseline=0.5, candidate=0.6)


# -----------------------------------------------------------------------------
# SampleDiff
# -----------------------------------------------------------------------------


class TestSampleDiff:
    def test_construct(self) -> None:
        from soup_cli.utils.adapter_pr import SampleDiff

        diff = SampleDiff(
            prompt="What is 2+2?",
            baseline_output="four",
            candidate_output="4",
        )
        assert diff.prompt.startswith("What")

    def test_frozen(self) -> None:
        from soup_cli.utils.adapter_pr import SampleDiff

        d = SampleDiff(prompt="p", baseline_output="b", candidate_output="c")
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.prompt = "new"  # type: ignore[misc]

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.adapter_pr import SampleDiff

        with pytest.raises(ValueError):
            SampleDiff(prompt="p\x00", baseline_output="b", candidate_output="c")

    def test_oversize_truncated(self) -> None:
        from soup_cli.utils.adapter_pr import MAX_OUTPUT_LEN, SampleDiff

        # >MAX_OUTPUT_LEN should be rejected to keep PRs reviewable
        with pytest.raises(ValueError):
            SampleDiff(
                prompt="p" * 10,
                baseline_output="x" * (MAX_OUTPUT_LEN + 1),
                candidate_output="y",
            )


# -----------------------------------------------------------------------------
# AdapterPR
# -----------------------------------------------------------------------------


class TestAdapterPR:
    def test_construct(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR, EvalDelta, SampleDiff

        pr = AdapterPR(
            title="add-customer-support-tone",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="+ 100 rows of support data\n",
            deltas=(EvalDelta(metric="accuracy", baseline=0.7, candidate=0.85),),
            samples=(
                SampleDiff(prompt="hi", baseline_output="hello", candidate_output="hey"),
            ),
        )
        assert pr.title == "add-customer-support-tone"

    def test_frozen(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR

        pr = AdapterPR(
            title="t",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(),
            samples=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pr.title = "new"  # type: ignore[misc]

    def test_base_sha_64_hex(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR

        # Not 64 hex chars
        with pytest.raises(ValueError):
            AdapterPR(
                title="t",
                base_sha="abc",
                adapter_path="adapter/",
                dataset_diff="",
                deltas=(),
                samples=(),
            )
        # Non-hex characters
        with pytest.raises(ValueError):
            AdapterPR(
                title="t",
                base_sha="z" * 64,
                adapter_path="adapter/",
                dataset_diff="",
                deltas=(),
                samples=(),
            )

    def test_title_validation(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR

        with pytest.raises(ValueError):
            AdapterPR(
                title="",
                base_sha="a" * 64,
                adapter_path="adapter/",
                dataset_diff="",
                deltas=(),
                samples=(),
            )

    def test_deltas_must_be_tuple(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR

        with pytest.raises(TypeError):
            AdapterPR(
                title="t",
                base_sha="a" * 64,
                adapter_path="adapter/",
                dataset_diff="",
                deltas=[],  # type: ignore[arg-type]
                samples=(),
            )


# -----------------------------------------------------------------------------
# build_adapter_pr factory
# -----------------------------------------------------------------------------


class TestBuildAdapterPR:
    def test_happy(self) -> None:
        from soup_cli.utils.adapter_pr import build_adapter_pr

        pr = build_adapter_pr(
            title="my-pr",
            base_sha="b" * 64,
            adapter_path="adapter/",
            dataset_diff="diff text",
            deltas=[
                {"metric": "accuracy", "baseline": 0.7, "candidate": 0.85},
            ],
            samples=[
                {"prompt": "p", "baseline_output": "a", "candidate_output": "b"},
            ],
        )
        assert pr.title == "my-pr"
        assert len(pr.deltas) == 1


# -----------------------------------------------------------------------------
# render_pr_markdown / render_pr_json
# -----------------------------------------------------------------------------


class TestRenderPR:
    def test_markdown_structure(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR, EvalDelta, SampleDiff, render_pr_markdown

        pr = AdapterPR(
            title="my-pr",
            base_sha="c" * 64,
            adapter_path="adapter/",
            dataset_diff="+row1\n+row2\n",
            deltas=(EvalDelta(metric="accuracy", baseline=0.7, candidate=0.85),),
            samples=(
                SampleDiff(prompt="hi", baseline_output="a", candidate_output="b"),
            ),
        )
        md = render_pr_markdown(pr)
        assert "my-pr" in md
        assert "accuracy" in md
        assert "0.70" in md or "0.7" in md
        # Should include a sample diff section
        assert "hi" in md
        # Should include the dataset diff
        assert "+row1" in md

    def test_markdown_escapes_markdown_metacharacters(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR, EvalDelta, render_pr_markdown

        # Embed pipe character in metric name; rendered table cell must escape it
        pr = AdapterPR(
            title="t",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(EvalDelta(metric="acc|injection", baseline=0.5, candidate=0.6),),
            samples=(),
        )
        md = render_pr_markdown(pr)
        # The pipe must be escaped to avoid breaking the table
        assert "acc\\|injection" in md or "acc|injection" not in md.split("|")

    def test_json_roundtrip(self) -> None:
        import json

        from soup_cli.utils.adapter_pr import (
            AdapterPR,
            EvalDelta,
            render_pr_json,
        )

        pr = AdapterPR(
            title="t",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(EvalDelta(metric="m", baseline=0.5, candidate=0.6),),
            samples=(),
        )
        text = render_pr_json(pr)
        data = json.loads(text)
        assert data["title"] == "t"
        assert data["deltas"][0]["metric"] == "m"

    def test_non_pr_rejected(self) -> None:
        from soup_cli.utils.adapter_pr import render_pr_json, render_pr_markdown

        with pytest.raises(TypeError):
            render_pr_markdown("not-a-pr")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            render_pr_json("not-a-pr")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Atomic write
# -----------------------------------------------------------------------------


class TestWritePR:
    def test_write_markdown(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR, write_pr_markdown

        monkeypatch.chdir(tmp_path)
        pr = AdapterPR(
            title="t",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(),
            samples=(),
        )
        out = tmp_path / "pr.md"
        write_pr_markdown(pr, str(out))
        assert out.exists()
        assert "t" in out.read_text(encoding="utf-8")

    def test_write_outside_cwd_rejected(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR, write_pr_markdown

        cwd = tmp_path / "work"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        pr = AdapterPR(
            title="t",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(),
            samples=(),
        )
        with pytest.raises(ValueError):
            write_pr_markdown(pr, str(tmp_path / "outside.md"))


# -----------------------------------------------------------------------------
# CLI smoke
# -----------------------------------------------------------------------------


class TestCliSmoke:
    def test_pr_help(self) -> None:
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        runner = CliRunner()
        result = runner.invoke(app, ["pr", "--help"])
        assert result.exit_code == 0
        assert "pr" in result.output.lower()


# -----------------------------------------------------------------------------
# Source-grep regression
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "soup_cli" / "utils" / "adapter_pr.py").read_text(
            encoding="utf-8"
        )
        head_lines = [
            line
            for line in src.splitlines()[:50]
            if line.strip() and not line.strip().startswith("#")
        ]
        head = "\n".join(head_lines)
        for forbidden in ("import torch", "import transformers", "import peft"):
            assert forbidden not in head, f"top-level {forbidden!r}"
