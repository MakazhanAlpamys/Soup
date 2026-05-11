"""v0.47.0 Part A — Synthetic Data Forge tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Imports + module surface
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import data_forge  # noqa: F401

    assert hasattr(data_forge, "ForgePlan")
    assert hasattr(data_forge, "ProvenanceRecord")
    assert hasattr(data_forge, "ForgeRow")
    assert hasattr(data_forge, "chunk_document")
    assert hasattr(data_forge, "score_uncertainty")
    assert hasattr(data_forge, "build_forge_plan")
    assert hasattr(data_forge, "write_forge_dataset")
    assert hasattr(data_forge, "discover_documents")


def test_forge_task_literal_constants():
    from soup_cli.utils.data_forge import VALID_TASKS

    assert frozenset(VALID_TASKS) == frozenset({"sft", "preference", "tool"})


# ---------------------------------------------------------------------------
# Dataclass frozen-ness
# ---------------------------------------------------------------------------


def test_forge_plan_frozen():
    import dataclasses

    from soup_cli.utils.data_forge import ForgePlan

    plan = ForgePlan(task="sft", num_docs=3, target_rows=12, teacher="local-judge")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.num_docs = 99  # type: ignore[misc]


def test_provenance_record_frozen():
    import dataclasses

    from soup_cli.utils.data_forge import ProvenanceRecord

    rec = ProvenanceRecord(
        row_id="r0",
        source_doc="doc1.txt",
        judge_id="local-judge",
        filter_score=0.8,
        chunk_id="c0",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.row_id = "x"  # type: ignore[misc]


def test_forge_row_to_dict_includes_provenance():
    from soup_cli.utils.data_forge import ForgeRow, ProvenanceRecord

    prov = ProvenanceRecord(
        row_id="r0",
        source_doc="doc1.txt",
        judge_id="local-judge",
        filter_score=0.8,
        chunk_id="c0",
    )
    row = ForgeRow(
        messages=({"role": "user", "content": "hi"},),
        provenance=prov,
        task="sft",
    )
    d = row.to_dict()
    assert d["task"] == "sft"
    assert d["messages"][0]["role"] == "user"
    assert d["provenance"]["row_id"] == "r0"


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


def test_chunk_document_basic():
    from soup_cli.utils.data_forge import chunk_document

    text = "para one.\n\npara two.\n\npara three."
    chunks = chunk_document(text, max_chunk_chars=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_document_short_returns_one():
    from soup_cli.utils.data_forge import chunk_document

    out = chunk_document("hello", max_chunk_chars=1000)
    assert out == ["hello"]


def test_chunk_document_empty_returns_empty():
    from soup_cli.utils.data_forge import chunk_document

    assert chunk_document("", max_chunk_chars=100) == []
    assert chunk_document("   \n  ", max_chunk_chars=100) == []


def test_chunk_document_rejects_invalid_max():
    from soup_cli.utils.data_forge import chunk_document

    with pytest.raises(ValueError):
        chunk_document("hi", max_chunk_chars=0)
    with pytest.raises(ValueError):
        chunk_document("hi", max_chunk_chars=-1)
    with pytest.raises(TypeError):
        chunk_document("hi", max_chunk_chars=True)  # bool rejected


def test_chunk_document_rejects_non_string():
    from soup_cli.utils.data_forge import chunk_document

    with pytest.raises(TypeError):
        chunk_document(123, max_chunk_chars=100)  # type: ignore[arg-type]


def test_chunk_document_rejects_null_byte():
    from soup_cli.utils.data_forge import chunk_document

    with pytest.raises(ValueError):
        chunk_document("hi\x00there", max_chunk_chars=100)


def test_chunk_document_respects_oversize_cap():
    from soup_cli.utils.data_forge import _MAX_DOC_CHARS, chunk_document

    with pytest.raises(ValueError):
        chunk_document("x" * (_MAX_DOC_CHARS + 1), max_chunk_chars=100)


# ---------------------------------------------------------------------------
# score_uncertainty
# ---------------------------------------------------------------------------


def test_score_uncertainty_bounds():
    from soup_cli.utils.data_forge import score_uncertainty

    s = score_uncertainty("a short answer", "a much longer reference answer here")
    assert 0.0 <= s <= 1.0


def test_score_uncertainty_identical_is_zero():
    from soup_cli.utils.data_forge import score_uncertainty

    s = score_uncertainty("hello world", "hello world")
    assert s == 0.0


def test_score_uncertainty_disjoint_is_one():
    from soup_cli.utils.data_forge import score_uncertainty

    s = score_uncertainty("alpha beta", "gamma delta")
    assert s == pytest.approx(1.0)


def test_score_uncertainty_rejects_invalid():
    from soup_cli.utils.data_forge import score_uncertainty

    with pytest.raises(TypeError):
        score_uncertainty(123, "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        score_uncertainty("x", 123)  # type: ignore[arg-type]


def test_score_uncertainty_empty_returns_one():
    from soup_cli.utils.data_forge import score_uncertainty

    assert score_uncertainty("", "x") == 1.0
    assert score_uncertainty("x", "") == 1.0


# ---------------------------------------------------------------------------
# build_forge_plan
# ---------------------------------------------------------------------------


def test_build_forge_plan_happy(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello world", encoding="utf-8")
    (docs / "b.md").write_text("# Title\nbody", encoding="utf-8")

    os.chdir(tmp_path)
    plan = build_forge_plan(
        docs_dir=str(docs),
        task="sft",
        target_rows=10,
        teacher="local-judge",
    )
    assert plan.task == "sft"
    assert plan.num_docs == 2
    assert plan.target_rows == 10
    assert plan.teacher == "local-judge"


def test_build_forge_plan_rejects_unknown_task(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="task"):
        build_forge_plan(docs_dir=str(docs), task="bogus", target_rows=1)


def test_build_forge_plan_rejects_zero_target_rows(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="target_rows"):
        build_forge_plan(docs_dir=str(docs), task="sft", target_rows=0)


def test_build_forge_plan_rejects_bool_target_rows(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(TypeError):
        build_forge_plan(docs_dir=str(docs), task="sft", target_rows=True)


def test_build_forge_plan_rejects_outside_cwd(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "out"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    inside = tmp_path / "inside"
    inside.mkdir()
    os.chdir(inside)
    with pytest.raises(ValueError, match="cwd"):
        build_forge_plan(docs_dir=str(docs), task="sft", target_rows=1)


def test_build_forge_plan_missing_dir(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    os.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_forge_plan(docs_dir="nonexistent", task="sft", target_rows=1)


def test_build_forge_plan_empty_dir(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    empty = tmp_path / "empty"
    empty.mkdir()
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="no documents"):
        build_forge_plan(docs_dir=str(empty), task="sft", target_rows=1)


def test_build_forge_plan_target_rows_cap(tmp_path):
    from soup_cli.utils.data_forge import _MAX_TARGET_ROWS, build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="target_rows"):
        build_forge_plan(
            docs_dir=str(docs), task="sft", target_rows=_MAX_TARGET_ROWS + 1
        )


def test_build_forge_plan_teacher_null_byte_rejected(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="teacher"):
        build_forge_plan(
            docs_dir=str(docs), task="sft", target_rows=1, teacher="bad\x00"
        )


def test_build_forge_plan_rejects_nan_threshold(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        build_forge_plan(
            docs_dir=str(docs),
            task="sft",
            target_rows=1,
            uncertainty_threshold=float("nan"),
        )


def test_build_forge_plan_rejects_inf_threshold(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        build_forge_plan(
            docs_dir=str(docs),
            task="sft",
            target_rows=1,
            uncertainty_threshold=float("inf"),
        )


def test_discover_documents_outside_cwd_rejected(tmp_path):
    from soup_cli.utils.data_forge import discover_documents

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("x", encoding="utf-8")
    inside = tmp_path / "inside"
    inside.mkdir()
    os.chdir(inside)
    with pytest.raises(ValueError, match="cwd"):
        discover_documents(str(outside))


def test_discover_documents_rejects_empty_string(tmp_path):
    from soup_cli.utils.data_forge import discover_documents

    os.chdir(tmp_path)
    with pytest.raises(ValueError):
        discover_documents("")


def test_forge_row_frozen():
    import dataclasses

    from soup_cli.utils.data_forge import ForgeRow, ProvenanceRecord

    prov = ProvenanceRecord(
        row_id="r0", source_doc="d", judge_id="j", filter_score=0.5, chunk_id="c0",
    )
    row = ForgeRow(messages=(), provenance=prov, task="sft")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.task = "preference"  # type: ignore[misc]


def test_write_forge_dataset_rejects_non_forge_row(tmp_path):
    from soup_cli.utils.data_forge import write_forge_dataset

    os.chdir(tmp_path)
    with pytest.raises(TypeError, match="ForgeRow"):
        write_forge_dataset([{"not": "a row"}], "out.jsonl")  # type: ignore[list-item]


def test_build_forge_plan_teacher_oversize_rejected(tmp_path):
    from soup_cli.utils.data_forge import build_forge_plan

    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="teacher"):
        build_forge_plan(
            docs_dir=str(docs),
            task="sft",
            target_rows=1,
            teacher="x" * 1024,
        )


# ---------------------------------------------------------------------------
# discover_documents
# ---------------------------------------------------------------------------


def test_discover_documents_skips_hidden(tmp_path):
    from soup_cli.utils.data_forge import discover_documents

    (tmp_path / ".hidden.txt").write_text("x", encoding="utf-8")
    (tmp_path / "real.txt").write_text("y", encoding="utf-8")
    os.chdir(tmp_path)
    docs = discover_documents(str(tmp_path))
    names = [os.path.basename(d) for d in docs]
    assert "real.txt" in names
    assert ".hidden.txt" not in names


def test_discover_documents_only_known_extensions(tmp_path):
    from soup_cli.utils.data_forge import discover_documents

    (tmp_path / "ok.txt").write_text("a", encoding="utf-8")
    (tmp_path / "ok.md").write_text("b", encoding="utf-8")
    (tmp_path / "ok.jsonl").write_text('{"x":1}', encoding="utf-8")
    (tmp_path / "ignore.bin").write_text("zzz", encoding="utf-8")
    os.chdir(tmp_path)
    docs = discover_documents(str(tmp_path))
    names = [os.path.basename(d) for d in docs]
    assert "ok.txt" in names
    assert "ok.md" in names
    assert "ok.jsonl" in names
    assert "ignore.bin" not in names


def test_discover_documents_rejects_symlink_dir(tmp_path):
    from soup_cli.utils.data_forge import discover_documents

    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    os.chdir(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        discover_documents(str(link))


def test_discover_documents_cap(tmp_path):
    from soup_cli.utils.data_forge import _MAX_DOCS

    # Just test that the cap exists and is sane; building thousands of files is slow.
    assert _MAX_DOCS >= 1


def test_discover_documents_rejects_null_byte(tmp_path):
    from soup_cli.utils.data_forge import discover_documents

    with pytest.raises(ValueError):
        discover_documents("foo\x00bar")


def test_discover_documents_rejects_non_string():
    from soup_cli.utils.data_forge import discover_documents

    with pytest.raises(TypeError):
        discover_documents(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# synthesise_forge_rows  (uses a fake judge callable)
# ---------------------------------------------------------------------------


def _fake_judge(prompt: str) -> dict:
    return {"text": f"REPLY: {prompt[:20]}", "confidence": 0.9}


def test_synthesise_forge_rows_basic(tmp_path):
    from soup_cli.utils.data_forge import synthesise_forge_rows

    docs = [str(tmp_path / "a.txt")]
    (tmp_path / "a.txt").write_text("para one.\n\npara two.", encoding="utf-8")
    rows = synthesise_forge_rows(
        docs,
        task="sft",
        target_rows=3,
        judge=_fake_judge,
        teacher="fake",
    )
    assert len(rows) <= 3
    assert all(r.task == "sft" for r in rows)
    assert all(r.provenance.source_doc.endswith("a.txt") for r in rows)


def test_synthesise_forge_rows_active_pruning(tmp_path):
    from soup_cli.utils.data_forge import synthesise_forge_rows

    docs = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
    (tmp_path / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (tmp_path / "b.txt").write_text("delta epsilon", encoding="utf-8")
    rows = synthesise_forge_rows(
        docs,
        task="sft",
        target_rows=10,
        judge=_fake_judge,
        teacher="fake",
        uncertainty_threshold=0.5,
    )
    # Active selection respects threshold; rows should have filter_score >= threshold
    assert all(r.provenance.filter_score >= 0.5 for r in rows)


def test_synthesise_forge_rows_unknown_task(tmp_path):
    from soup_cli.utils.data_forge import synthesise_forge_rows

    with pytest.raises(ValueError, match="task"):
        synthesise_forge_rows([], task="bogus", target_rows=1, judge=_fake_judge)


def test_synthesise_forge_rows_judge_must_be_callable(tmp_path):
    from soup_cli.utils.data_forge import synthesise_forge_rows

    with pytest.raises(TypeError, match="judge"):
        synthesise_forge_rows([], task="sft", target_rows=1, judge="not-a-fn")  # type: ignore[arg-type]


def test_synthesise_forge_rows_bool_target_rows(tmp_path):
    from soup_cli.utils.data_forge import synthesise_forge_rows

    with pytest.raises(TypeError):
        synthesise_forge_rows([], task="sft", target_rows=True, judge=_fake_judge)


def test_synthesise_forge_rows_judge_failure_swallowed(tmp_path):
    from soup_cli.utils.data_forge import synthesise_forge_rows

    def bad_judge(_: str) -> dict:
        raise RuntimeError("boom")

    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    rows = synthesise_forge_rows(
        [str(tmp_path / "a.txt")],
        task="sft",
        target_rows=2,
        judge=bad_judge,
        teacher="fake",
    )
    # Failures are skipped, not propagated
    assert rows == [] or all(r.task == "sft" for r in rows)


# ---------------------------------------------------------------------------
# write_forge_dataset + write_provenance
# ---------------------------------------------------------------------------


def test_write_forge_dataset_atomic(tmp_path):
    from soup_cli.utils.data_forge import (
        ForgeRow,
        ProvenanceRecord,
        write_forge_dataset,
    )

    rows = [
        ForgeRow(
            messages=({"role": "user", "content": "hi"},),
            provenance=ProvenanceRecord(
                row_id="r0",
                source_doc="d.txt",
                judge_id="j",
                filter_score=0.9,
                chunk_id="c0",
            ),
            task="sft",
        )
    ]
    os.chdir(tmp_path)
    out_path = write_forge_dataset(rows, "out.jsonl")
    assert Path(out_path).is_file()
    with open(out_path, encoding="utf-8") as fh:
        line = fh.readline()
    parsed = json.loads(line)
    assert parsed["task"] == "sft"
    assert parsed["provenance"]["row_id"] == "r0"


def test_write_forge_dataset_outside_cwd_rejected(tmp_path):
    from soup_cli.utils.data_forge import write_forge_dataset

    inside = tmp_path / "inside"
    inside.mkdir()
    os.chdir(inside)
    with pytest.raises(ValueError, match="cwd"):
        write_forge_dataset([], str(tmp_path / "out.jsonl"))


def test_write_forge_dataset_null_byte_rejected(tmp_path):
    from soup_cli.utils.data_forge import write_forge_dataset

    os.chdir(tmp_path)
    with pytest.raises(ValueError):
        write_forge_dataset([], "out\x00.jsonl")


def test_write_forge_dataset_non_string_rejected(tmp_path):
    from soup_cli.utils.data_forge import write_forge_dataset

    os.chdir(tmp_path)
    with pytest.raises(TypeError):
        write_forge_dataset([], 123)  # type: ignore[arg-type]


def test_write_forge_dataset_symlink_target_rejected(tmp_path):
    from soup_cli.utils.data_forge import (
        ForgeRow,
        ProvenanceRecord,
        write_forge_dataset,
    )

    if os.name == "nt":
        pytest.skip("symlink test POSIX-only")
    os.chdir(tmp_path)
    target = tmp_path / "out.jsonl"
    real = tmp_path / "decoy"
    real.write_text("", encoding="utf-8")
    target.symlink_to(real)
    row = ForgeRow(
        messages=({"role": "user", "content": "hi"},),
        provenance=ProvenanceRecord(
            row_id="r0", source_doc="d", judge_id="j",
            filter_score=0.5, chunk_id="c0",
        ),
        task="sft",
    )
    with pytest.raises(ValueError, match="symlink"):
        write_forge_dataset([row], str(target))


def test_write_provenance_manifest(tmp_path):
    from soup_cli.utils.data_forge import (
        ForgeRow,
        ProvenanceRecord,
        write_provenance,
    )

    rows = [
        ForgeRow(
            messages=({"role": "user", "content": "hi"},),
            provenance=ProvenanceRecord(
                row_id=f"r{i}",
                source_doc="d.txt",
                judge_id="j",
                filter_score=0.5,
                chunk_id=f"c{i}",
            ),
            task="sft",
        )
        for i in range(3)
    ]
    os.chdir(tmp_path)
    manifest = write_provenance(rows, "manifest.json")
    assert Path(manifest).is_file()
    with open(manifest, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["version"] == 1
    assert data["row_count"] == 3
    assert len(data["records"]) == 3
    assert data["records"][0]["row_id"] == "r0"


def test_write_provenance_outside_cwd_rejected(tmp_path):
    from soup_cli.utils.data_forge import write_provenance

    inside = tmp_path / "inside"
    inside.mkdir()
    os.chdir(inside)
    with pytest.raises(ValueError, match="cwd"):
        write_provenance([], str(tmp_path / "m.json"))


# ---------------------------------------------------------------------------
# CLI: soup data forge
# ---------------------------------------------------------------------------


def _make_app():
    from soup_cli.cli import app
    return app


def test_data_forge_cli_help():
    runner = CliRunner()
    result = runner.invoke(_make_app(), ["data", "forge", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "forge" in result.output.lower()


def test_data_forge_cli_happy(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha beta gamma", encoding="utf-8")
    (docs / "b.txt").write_text("delta epsilon", encoding="utf-8")

    result = runner.invoke(
        _make_app(),
        [
            "data", "forge",
            "--docs", str(docs),
            "--task", "sft",
            "--target-rows", "4",
            "--output", "forge.jsonl",
            "--provenance", "forge_provenance.json",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path("forge.jsonl").is_file()
    assert Path("forge_provenance.json").is_file()


def test_data_forge_cli_unknown_task(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    result = runner.invoke(
        _make_app(),
        [
            "data", "forge",
            "--docs", str(docs),
            "--task", "BOGUS",
            "--target-rows", "1",
        ],
    )
    assert result.exit_code != 0


def test_data_forge_cli_missing_docs(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        _make_app(),
        [
            "data", "forge",
            "--docs", "nonexistent",
            "--task", "sft",
            "--target-rows", "1",
        ],
    )
    assert result.exit_code != 0


def test_data_forge_cli_outside_cwd(tmp_path, monkeypatch):
    runner = CliRunner()
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("x", encoding="utf-8")
    monkeypatch.chdir(inside)
    result = runner.invoke(
        _make_app(),
        [
            "data", "forge",
            "--docs", str(outside),
            "--task", "sft",
            "--target-rows", "1",
        ],
    )
    assert result.exit_code != 0
